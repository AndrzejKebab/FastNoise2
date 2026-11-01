// Builds against both the base commit and the fork; uses public API only
#include <FastNoise/FastNoise.h>
#include <FastNoise/Metadata.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
    struct FeatureSetName
    {
        FastSIMD::FeatureSet featureSet;
        const char* name;
    };

    constexpr FeatureSetName kFeatureSets[] = {
        { FastSIMD::FeatureSet::SSE2, "SSE2" },
        { FastSIMD::FeatureSet::SSE41, "SSE41" },
        { FastSIMD::FeatureSet::AVX2, "AVX2" },
        { FastSIMD::FeatureSet::AVX512, "AVX512" },
        { FastSIMD::FeatureSet::NEON, "NEON" },
        { FastSIMD::FeatureSet::AARCH64, "AARCH64" },
    };

    // Nodes added by the fork are not part of the base comparison
    bool IsForkOnlyNode( const FastNoise::Metadata* metadata )
    {
        return std::strcmp( metadata->name, "FractalDamped" ) == 0;
    }

    const char* FeatureSetString( FastSIMD::FeatureSet featureSet )
    {
        for( const FeatureSetName& entry: kFeatureSets )
        {
            if( entry.featureSet == featureSet )
            {
                return entry.name;
            }
        }
        return "Unknown";
    }

    // The distinct feature sets FastNoise creates nodes with on this CPU
    std::vector<FastSIMD::FeatureSet> AvailableFeatureSets()
    {
        // An explicit level is not checked against the CPU, only the default one is
        FastSIMD::FeatureSet best = FastNoise::New<FastNoise::Constant>()->GetActiveFeatureSet();

        std::vector<FastSIMD::FeatureSet> result;
        for( const FeatureSetName& entry: kFeatureSets )
        {
            auto node = entry.featureSet <= best ? FastNoise::New<FastNoise::Constant>( entry.featureSet ) : nullptr;
            if( node && node->GetActiveFeatureSet() == entry.featureSet )
            {
                result.push_back( entry.featureSet );
            }
        }
        return result;
    }

    const FastNoise::Metadata* FindMetadata( const char* name )
    {
        for( const FastNoise::Metadata* metadata: FastNoise::Metadata::GetAll() )
        {
            if( std::strcmp( metadata->name, name ) == 0 )
            {
                return metadata;
            }
        }
        return nullptr;
    }

    void WriteSamples( std::ofstream& out, const FastNoise::Generator* generator )
    {
        constexpr int k2D = 48;
        constexpr int k3D = 12;
        std::vector<float> noise( k2D * k2D );

        generator->GenUniformGrid2D( noise.data(), -37.5f, 12.25f, k2D, k2D, 3.1f, 2.7f, 1337 );
        out.write( reinterpret_cast<const char*>( noise.data() ), noise.size() * sizeof( float ) );

        noise.resize( k3D * k3D * k3D );
        generator->GenUniformGrid3D( noise.data(), 5.5f, -80.0f, 40.125f, k3D, k3D, k3D, 4.3f, 3.9f, 5.1f, -4242 );
        out.write( reinterpret_cast<const char*>( noise.data() ), noise.size() * sizeof( float ) );

        float xs[] = { 0.0f, 1.0e6f, -1.0e6f, 123.456f, -98765.4f, 0.5f, 33333.3f };
        float ys[] = { 0.0f, -1.0e6f, 1.0e6f, 654.321f, 4567.8f, -0.5f, -77777.7f };
        float zs[] = { 0.0f, 1.0e6f, 1.0e6f, -11.11f, 222.2f, 0.25f, 0.0f };
        constexpr int kPoints = sizeof( xs ) / sizeof( float );
        float points[kPoints];
        generator->GenPositionArray2D( points, kPoints, xs, ys, 0.0f, 0.0f, 7 );
        out.write( reinterpret_cast<const char*>( points ), sizeof( points ) );
        generator->GenPositionArray3D( points, kPoints, xs, ys, zs, 0.0f, 0.0f, 0.0f, 7 );
        out.write( reinterpret_cast<const char*>( points ), sizeof( points ) );
    }

    std::vector<std::string> ReadLines( const std::string& path )
    {
        std::vector<std::string> lines;
        std::ifstream file( path );
        std::string line;
        while( std::getline( file, line ) )
        {
            while( !line.empty() && ( line.back() == '\r' || line.back() == ' ' ) )
            {
                line.pop_back();
            }
            if( !line.empty() && line[0] != '#' )
            {
                lines.push_back( line );
            }
        }
        return lines;
    }

    // Every node's metadata name, in node ID order
    int DumpMetadata( const char* outPath )
    {
        std::ofstream out( outPath );
        const auto& all = FastNoise::Metadata::GetAll();
        out << "count " << all.size() << "\n";
        for( const FastNoise::Metadata* metadata: all )
        {
            out << (int)metadata->id << " " << metadata->name << "\n";
        }
        return 0;
    }

    int DumpSamples( const char* fixturesPath, const char* outPath )
    {
        std::ofstream out( outPath, std::ios::binary );
        std::vector<std::string> fixtures = ReadLines( fixturesPath );
        int failures = 0;

        for( FastSIMD::FeatureSet featureSet: AvailableFeatureSets() )
        {
            std::cout << "Feature set: " << FeatureSetString( featureSet ) << std::endl;
            out << FeatureSetString( featureSet );

            for( const FastNoise::Metadata* metadata: FastNoise::Metadata::GetAll() )
            {
                if( IsForkOnlyNode( metadata ) )
                {
                    continue;
                }

                FastNoise::SmartNode<> node = metadata->CreateNode( featureSet );
                bool warpSource = std::strncmp( metadata->name, "DomainWarpFractal", 17 ) == 0;
                for( const auto& lookup: metadata->memberNodeLookups )
                {
                    FastNoise::SmartNode<> source = FastNoise::New<FastNoise::Simplex>( featureSet );
                    if( warpSource )
                    {
                        auto warp = FastNoise::New<FastNoise::DomainWarpSimplex>( featureSet );
                        warp->SetSource( source );
                        source = warp;
                    }
                    lookup.setFunc( node.get(), source );
                }
                out << metadata->name;
                WriteSamples( out, node.get() );
            }

            for( size_t i = 0; i < fixtures.size(); i++ )
            {
                FastNoise::SmartNode<> node = FastNoise::NewFromEncodedNodeTree( fixtures[i].c_str(), featureSet );
                if( !node )
                {
                    std::cerr << "Fixture " << i << " failed to decode" << std::endl;
                    failures++;
                    continue;
                }
                out << "fixture" << i;
                WriteSamples( out, node.get() );
            }
        }

        std::cout << fixtures.size() << " fixtures" << std::endl;
        return failures ? 1 : 0;
    }

    // Value for a variable that differs from the default and stays inside its range
    FastNoise::Metadata::MemberVariable::ValueUnion NonDefault( const FastNoise::Metadata::MemberVariable& variable )
    {
        using Var = FastNoise::Metadata::MemberVariable;
        Var::ValueUnion value = variable.valueDefault;

        switch( variable.type )
        {
        case Var::EFloat:
            value.f = variable.valueDefault.f + 0.375f;
            if( variable.valueMax.f > variable.valueMin.f && value.f > variable.valueMax.f )
            {
                value.f = variable.valueDefault.f - 0.375f;
            }
            break;
        case Var::EInt:
            value.i = variable.valueDefault.i + 1;
            if( variable.valueMax.i > variable.valueMin.i && value.i > variable.valueMax.i )
            {
                value.i = variable.valueDefault.i - 1;
            }
            break;
        case Var::EEnum:
            value.i = (int)variable.enumNames.size() - 1;
            if( value.i == variable.valueDefault.i )
            {
                value.i = 0;
            }
            break;
        }
        return value;
    }

    using NodeDataStore = std::vector<std::unique_ptr<FastNoise::NodeData>>;

    FastNoise::NodeData* AddNode( NodeDataStore& store, const char* name )
    {
        store.emplace_back( std::make_unique<FastNoise::NodeData>( FindMetadata( name ) ) );
        return store.back().get();
    }

    std::string MakeTree( const FastNoise::Metadata* metadata, bool nonDefault )
    {
        NodeDataStore store;
        store.emplace_back( std::make_unique<FastNoise::NodeData>( metadata ) );
        FastNoise::NodeData* root = store.back().get();

        // DomainWarpFractal nodes take a DomainWarp source
        bool warpSource = std::strncmp( metadata->name, "DomainWarpFractal", 17 ) == 0;

        for( size_t i = 0; i < root->nodeLookups.size(); i++ )
        {
            FastNoise::NodeData* source = AddNode( store, warpSource ? "DomainWarpSimplex" : nonDefault ? "Perlin"
                                                                                                        : "Simplex" );
            if( warpSource )
            {
                source->nodeLookups[0] = AddNode( store, nonDefault ? "Perlin" : "Simplex" );
            }
            if( nonDefault )
            {
                for( size_t v = 0; v < source->variables.size(); v++ )
                {
                    source->variables[v] = NonDefault( source->metadata->memberVariables[v] );
                }
            }
            root->nodeLookups[i] = source;
        }

        if( nonDefault )
        {
            for( size_t v = 0; v < root->variables.size(); v++ )
            {
                root->variables[v] = NonDefault( metadata->memberVariables[v] );
            }
            for( size_t h = 0; h < root->hybrids.size(); h++ )
            {
                // Odd hybrids get a node, even ones a non-default constant, so both encodings are covered
                if( h % 2 == 0 )
                {
                    FastNoise::NodeData* source = AddNode( store, "Value" );
                    source->variables[0] = NonDefault( source->metadata->memberVariables[0] );
                    root->hybrids[h].first = source;
                }
                else
                {
                    root->hybrids[h].second = metadata->memberHybrids[h].valueDefault + 0.25f;
                }
            }
        }

        return FastNoise::Metadata::SerialiseNodeData( root );
    }

    int GenerateFixtures( const char* outPath )
    {
        std::ofstream out( outPath );
        out << "# Encoded node trees from the base build, one per line\n";
        out << "# Default and non-default trees for every node type, then hand-built trees\n";
        int failures = 0;

        for( const FastNoise::Metadata* metadata: FastNoise::Metadata::GetAll() )
        {
            for( bool nonDefault: { false, true } )
            {
                std::string encoded = MakeTree( metadata, nonDefault );
                if( encoded.empty() || !FastNoise::NewFromEncodedNodeTree( encoded.c_str() ) )
                {
                    std::cerr << metadata->name << ( nonDefault ? " non-default" : " default" ) << " tree failed" << std::endl;
                    failures++;
                    continue;
                }
                out << encoded << "\n";
            }
        }

        // A deeper tree mixing fractals, warps, blends and modifiers
        {
            NodeDataStore store;
            FastNoise::NodeData* fbm = AddNode( store, "FractalFBm" );
            FastNoise::NodeData* warp = AddNode( store, "DomainWarpFractalProgressive" );
            FastNoise::NodeData* warpSource = AddNode( store, "DomainWarpSimplex" );
            FastNoise::NodeData* cellular = AddNode( store, "CellularDistance" );
            FastNoise::NodeData* remap = AddNode( store, "Remap" );
            FastNoise::NodeData* blend = AddNode( store, "Fade" );
            FastNoise::NodeData* value = AddNode( store, "Value" );
            FastNoise::NodeData* perlin = AddNode( store, "Perlin" );

            warpSource->nodeLookups[0] = cellular;
            warp->nodeLookups[0] = warpSource;
            remap->nodeLookups[0] = perlin;
            blend->nodeLookups[0] = warp;
            blend->nodeLookups[1] = remap;
            blend->hybrids[0].first = value;
            fbm->nodeLookups[0] = blend;
            fbm->variables[0] = 5;
            fbm->hybrids[1].second = 0.6f;

            std::string encoded = FastNoise::Metadata::SerialiseNodeData( fbm );
            out << encoded << "\n";
        }

        return failures ? 1 : 0;
    }
} // namespace

int main( int argc, char** argv )
{
    if( argc == 3 && std::strcmp( argv[1], "metadata" ) == 0 )
    {
        return DumpMetadata( argv[2] );
    }
    if( argc == 4 && std::strcmp( argv[1], "samples" ) == 0 )
    {
        return DumpSamples( argv[2], argv[3] );
    }
    if( argc == 3 && std::strcmp( argv[1], "fixtures" ) == 0 )
    {
        return GenerateFixtures( argv[2] );
    }

    std::cerr << "Usage:\n"
                 "  FastNoiseRegressionDump metadata <out.txt>\n"
                 "  FastNoiseRegressionDump samples <fixtures.txt> <out.bin>\n"
                 "  FastNoiseRegressionDump fixtures <fixtures.txt>\n";
    return 2;
}
