#pragma once
#include <FastNoise/FastNoise.h>
#include <FastNoise/Metadata.h>
#include <FastSIMD/DispatchClass.h>

// Test-only node: reads value-plus-gradient results through the SIMD interface the library nodes use
class GradientProbe : public virtual FastNoise::Generator
{
public:
    enum class Formula
    {
        Simplex,
        Perlin,
        SuperSimplex,
        Value,
    };

    const FastNoise::Metadata& GetMetadata() const override
    {
        return FastNoise::Metadata::Get<FastNoise::Constant>();
    }

    void SetTarget( FastNoise::SmartNodeArg<> node )
    {
        this->SetSourceMemberVariable( mTarget, node );
    }

    // The target's GenGradient, positions interleaved per point; false when the target has no gradient
    virtual bool NodeGradient( int dims, int seed, const float* positions, int count, float* values, float* gradients ) const = 0;

    // The formula's ValueGradient on unit-lattice positions; values go through the default [-1, 1] ScaleOutput when scaleOutput is set
    virtual void FormulaGradient( Formula formula, int dims, int seed, const float* positions, int count, float* values, float* gradients, bool scaleOutput ) const = 0;

    // The formula's native output bound, the value ScaleOutput maps to 1
    virtual float FormulaBound( Formula formula, int dims ) const = 0;

protected:
    FastNoise::GeneratorSource mTarget;
};
