// Builds against both the base commit and the fork; uses public API only
#include <FastNoise/FastNoise.h>
#include <FastNoise/FastNoise_C.h>
#include <FastNoise/Metadata.h>

#include "../../tools/NodeEditor/util/DemoNodeTrees.inl"

#include <cmath>
#include <cstdint>
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
    // Through the C API, which is what consumers read node IDs from
    int DumpMetadata( const char* outPath )
    {
        std::ofstream out( outPath );
        int count = fnGetMetadataCount();
        out << "count " << count << "\n";
        for( int id = 0; id < count; id++ )
        {
            out << id << " " << fnGetMetadataName( id ) << "\n";
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
                std::vector<std::unique_ptr<FastNoise::NodeData>> store;
                FastNoise::NodeData* tree = FastNoise::Metadata::DeserialiseNodeData( fixtures[i].c_str(), store );
                if( !node || !tree )
                {
                    std::cerr << "Fixture " << i << " failed to decode" << std::endl;
                    failures++;
                    continue;
                }
                // The decoded tree, re-encoded, so the comparison covers tree identity as well as output
                out << "fixture" << i << FastNoise::Metadata::SerialiseNodeData( tree );
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
                // Even hybrids get a node, odd ones a non-default constant, so both encodings are covered
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

        out << "# NodeEditor demo trees\n";
        for( const auto& demo: gDemoNodeTrees )
        {
            out << demo[1] << "\n";
        }

        return failures ? 1 : 0;
    }
} // namespace

namespace
{
    void WriteSection( std::ofstream& out, const std::string& name, const std::vector<float>& values )
    {
        uint32_t nameLength = (uint32_t)name.size(), count = (uint32_t)values.size();
        out.write( reinterpret_cast<const char*>( &nameLength ), sizeof( nameLength ) );
        out.write( name.data(), nameLength );
        out.write( reinterpret_cast<const char*>( &count ), sizeof( count ) );
        out.write( reinterpret_cast<const char*>( values.data() ), count * sizeof( float ) );
    }

    bool ReadSection( std::ifstream& in, std::string& name, std::vector<float>& values )
    {
        uint32_t nameLength, count;
        if( !in.read( reinterpret_cast<char*>( &nameLength ), sizeof( nameLength ) ) )
        {
            return false;
        }
        name.resize( nameLength );
        in.read( &name[0], nameLength );
        in.read( reinterpret_cast<char*>( &count ), sizeof( count ) );
        values.resize( count );
        in.read( reinterpret_cast<char*>( values.data() ), count * sizeof( float ) );
        return (bool)in;
    }

    // FractalDamped output for every source, mode and dimension count, to measure drift between two builds
    int DumpDamped( const char* outPath )
    {
        const FastNoise::Metadata* damped = FindMetadata( "FractalDamped" );
        if( !damped )
        {
            std::cerr << "This build has no FractalDamped" << std::endl;
            return 1;
        }
        auto variableIndex = [&]( const char* name ) {
            for( size_t i = 0; i < damped->memberVariables.size(); i++ )
            {
                if( std::strcmp( damped->memberVariables[i].name, name ) == 0 )
                {
                    return i;
                }
            }
            return (size_t)-1;
        };

        std::ofstream out( outPath, std::ios::binary );
        for( const char* sourceName: { "Simplex", "Perlin", "SuperSimplex", "Value" } )
        {
            for( int mode: { 0, 1 } )
            {
                NodeDataStore store;
                store.emplace_back( std::make_unique<FastNoise::NodeData>( damped ) );
                FastNoise::NodeData* root = store.back().get();
                root->nodeLookups[0] = AddNode( store, sourceName );
                root->variables[variableIndex( "Octaves" )] = 5;
                root->variables[variableIndex( "Mode" )] = mode;
                FastNoise::SmartNode<> node = FastNoise::NewFromEncodedNodeTree( FastNoise::Metadata::SerialiseNodeData( root ).c_str() );

                std::string name = std::string( sourceName ) + ( mode ? " Approximate " : " Analytic " );
                std::vector<float> values( 96 * 96 );
                node->GenUniformGrid2D( values.data(), -150, 75, 96, 96, 3.1f, 3.1f, 1337 );
                WriteSection( out, name + "2D", values );
                values.resize( 20 * 20 * 20 );
                node->GenUniformGrid3D( values.data(), -40, 10, 90, 20, 20, 20, 4.3f, 4.3f, 4.3f, 1337 );
                WriteSection( out, name + "3D", values );
                values.resize( 9 * 9 * 9 * 9 );
                node->GenUniformGrid4D( values.data(), 20, -60, 5, 33, 9, 9, 9, 9, 5.7f, 5.7f, 5.7f, 5.7f, 1337 );
                WriteSection( out, name + "4D", values );
            }
        }
        return 0;
    }

    int CompareDumps( const char* beforePath, const char* afterPath )
    {
        std::ifstream before( beforePath, std::ios::binary ), after( afterPath, std::ios::binary );
        std::string nameA, nameB;
        std::vector<float> a, b;
        std::printf( "%-28s %12s %12s %12s\n", "section", "max |diff|", "mean |diff|", "max |value|" );
        while( ReadSection( before, nameA, a ) && ReadSection( after, nameB, b ) )
        {
            if( nameA != nameB || a.size() != b.size() )
            {
                std::cerr << "Dumps differ in layout at " << nameA << std::endl;
                return 1;
            }
            double maxDiff = 0, sumDiff = 0, maxValue = 0;
            for( size_t i = 0; i < a.size(); i++ )
            {
                double diff = std::abs( (double)a[i] - b[i] );
                maxDiff = std::max( maxDiff, diff );
                sumDiff += diff;
                maxValue = std::max( maxValue, (double)std::abs( a[i] ) );
            }
            std::printf( "%-28s %12.3g %12.3g %12.3g\n", nameA.c_str(), maxDiff, sumDiff / a.size(), maxValue );
        }
        return 0;
    }
} // namespace

int main( int argc, char** argv )
{
    if( argc == 3 && std::strcmp( argv[1], "damped" ) == 0 )
    {
        return DumpDamped( argv[2] );
    }
    if( argc == 4 && std::strcmp( argv[1], "compare" ) == 0 )
    {
        return CompareDumps( argv[2], argv[3] );
    }
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
                 "  FastNoiseRegressionDump fixtures <fixtures.txt>\n"
                 "  FastNoiseRegressionDump damped <out.bin>\n"
                 "  FastNoiseRegressionDump compare <before.bin> <after.bin>\n";
    return 2;
}
