#pragma once
#include "Generator.h"

namespace FastNoise
{
    template<typename T = Generator>
    class Fractal : public virtual Generator
    {
    public:
        void SetSource( SmartNodeArg<T> gen ) { this->SetSourceMemberVariable( mSource, gen ); }
        void SetGain( float value ) { mGain = value; }
        void SetGain( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mGain, gen ); }
        void SetWeightedStrength( float value ) { mWeightedStrength = value; } 
        void SetWeightedStrength( SmartNodeArg<> gen ) { this->SetSourceMemberVariable( mWeightedStrength, gen ); }
        void SetOctaveCount( int value ) { mOctaves = value; }
        void SetLacunarity( float value ) { mLacunarity = value; } 

    protected:
        GeneratorSourceT<T> mSource;
        HybridSource mGain = 0.5f;
        HybridSource mWeightedStrength = 0.0f;

        int   mOctaves = 3;
        float mLacunarity = 2.0f;
    };

#ifdef FASTNOISE_METADATA
    template<typename T>
    struct MetadataT<Fractal<T>> : MetadataT<Generator>
    {
        MetadataT( NameDesc sourceName = "Source", bool addGroup = true )
        {
            if( addGroup )
            {
                groups.push_back( "Fractal" );
            }
            this->AddGeneratorSource( sourceName, &Fractal<T>::SetSource );
            this->AddHybridSource( { "Gain",
                "Multiplier for how much each octave contributes to the final result\n"
                "Lower values create smoother, more uniform noise\n"
                "Higher values preserve more detail from smaller octaves" },
                0.5f, &Fractal<T>::SetGain, &Fractal<T>::SetGain );

            this->AddHybridSource( { "Weighted Strength",
                "Scales the contribution of higher octaves based on the output value of previous octaves\n"
                "Low values from lower octaves reduces contribution from higher octaves\n"
                "Setting this higher creates smoother output in low areas while retaining octave details in high areas" },
                0.0f, &Fractal<T>::SetWeightedStrength, &Fractal<T>::SetWeightedStrength );

            this->AddVariable( { "Octaves",
                "Number of noise layers combined together\n"
                "More octaves = more detail but slower generation\n"
                "Each octave adds finer detail at higher frequencies" },
                3, &Fractal<T>::SetOctaveCount, 2, 16 );

            this->AddVariable( { "Lacunarity",
                "How much the frequency increases between octaves\n"
                "2.0 = Each octave is twice as detailed as the previous\n"
                "Higher values create more contrast between large and small features" },
                2.0f, &Fractal<T>::SetLacunarity );
        }
    };
#endif

    class FractalFBm : public virtual Fractal<>
    {
    public:
        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalFBm> : MetadataT<Fractal<>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;
        
        MetadataT() : MetadataT<Fractal<>>()
        {
            description = 
                "Fractional Brownian Motion - Builds up realistic natural textures\n"
                "Combines multiple octaves where each adds progressively finer detail\n"
                "Creates smooth, flowing patterns ideal for terrain, clouds, and organic textures\n"
                "Each octave's contribution decreases by the Gain amount";
        }
    };
#endif

    class FractalRidged : public virtual Fractal<>
    {
    public:
        const Metadata& GetMetadata() const override;
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalRidged> : MetadataT<Fractal<>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;
        
        MetadataT() : MetadataT<Fractal<>>()
        {
            description = 
                "Creates sharp ridges and valleys by inverting and combining noise octaves\n"
                "Produces dramatic mountain-like terrain with defined peaks and canyons\n"
                "Higher octaves sharpen the ridges rather than adding smooth detail\n"
                "Perfect for creating realistic mountain ranges and rocky formations";
        }
    };
#endif

    class FractalDamped : public virtual Fractal<>
    {
    public:
        const Metadata& GetMetadata() const override;

        enum class Mode
        {
            Analytic,
            Approximate,
        };

        enum class SlopeFormula
        {
            Auto,
            Simplex,
            Perlin,
            SuperSimplex,
            Value,
        };

        void SetRotation( float degrees )
        {
            mRotation = degrees;
            mRotationCos = std::cos( degrees * ( 3.14159265358979323846f / 180.0f ) );
            mRotationSin = std::sin( degrees * ( 3.14159265358979323846f / 180.0f ) );
        }
        void SetMode( Mode value )
        {
            mMode = value;
        }
        void SetSlopeFormula( SlopeFormula value )
        {
            mSlopeFormula = value;
        }
        void SetSlopeFeatureScale( float value )
        {
            mSlopeFeatureScale = value;
        }
        void SetSlopeSeedOffset( int value )
        {
            mSlopeSeedOffset = value;
        }

    protected:
        float mRotation = 36.87f;
        float mRotationCos = std::cos( 36.87f * ( 3.14159265358979323846f / 180.0f ) );
        float mRotationSin = std::sin( 36.87f * ( 3.14159265358979323846f / 180.0f ) );
        Mode mMode = Mode::Analytic;
        SlopeFormula mSlopeFormula = SlopeFormula::Auto;
        float mSlopeFeatureScale = 100.0f;
        int mSlopeSeedOffset = 0;
    };

    constexpr static const char* kFractalDampedMode_Strings[] = {
        "Analytic",
        "Approximate",
    };

    constexpr static const char* kFractalDampedSlopeFormula_Strings[] = {
        "Auto",
        "Simplex",
        "Perlin",
        "SuperSimplex",
        "Value",
    };

#ifdef FASTNOISE_METADATA
    template<>
    struct MetadataT<FractalDamped> : MetadataT<Fractal<>>
    {
        SmartNode<> CreateNode( FastSIMD::FeatureSet ) const override;

        MetadataT() : MetadataT<Fractal<>>()
        {
            this->AddVariable( { "Rotation",
                                 "Rotation in degrees between octaves\n"
                                 "Turns the first two axes, also in 3D and 4D\n"
                                 "36.87 is the rotation of the published method" },
                               36.87f, &FractalDamped::SetRotation );

            this->AddVariableEnum( { "Mode",
                                     "Where each octave's gradient comes from\n"
                                     "Analytic: exact gradient from the source or the Slope Formula\n"
                                     "Approximate: central differences of the source, 2N extra samples per octave" },
                                   FractalDamped::Mode::Analytic, &FractalDamped::SetMode, kFractalDampedMode_Strings );

            this->AddVariableEnum( { "Slope Formula",
                                     "Analytic mode only\n"
                                     "Auto: the source's own gradient, or Approximate when the source has none\n"
                                     "Simplex/Perlin/SuperSimplex/Value: that noise's gradient at the Slope Feature Scale and Slope Seed Offset, normalised to a [-1, 1] output range" },
                                   FractalDamped::SlopeFormula::Auto, &FractalDamped::SetSlopeFormula, kFractalDampedSlopeFormula_Strings );

            this->AddVariable( { "Slope Feature Scale",
                                 "Feature scale used by an explicit Slope Formula\n"
                                 "Also the cell size for Approximate when the source has no Feature Scale" },
                               100.0f, &FractalDamped::SetSlopeFeatureScale, 0.f, 0.f, 0.25f );

            this->AddVariable( { "Slope Seed Offset",
                                 "Seed offset used by an explicit Slope Formula" },
                               0, &FractalDamped::SetSlopeSeedOffset );

            description =
                "Fractal sum where each octave is damped by the slope the octaves before it built up\n"
                "Steep ground keeps its large forms and loses its fine roughness\n"
                "Derivative-damped terrain after Inigo Quilez\n"
                "The slope sum is not rotated back, so it is not the slope of the output\n"
                "In 3D and 4D, Rotation turns the first two axes and the slope sum uses every axis\n"
                "In 4D, Analytic falls back to Approximate";
        }
    };
#endif
}
