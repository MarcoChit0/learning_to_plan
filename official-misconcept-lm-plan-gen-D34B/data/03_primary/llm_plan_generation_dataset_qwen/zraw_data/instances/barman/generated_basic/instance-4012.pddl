(define (problem prob)
 (:domain barman)
 (:objects 
      shaker195 - shaker
      left right - hand
      shot446 shot121 shot309 shot30 - shot
      ingredient151 ingredient270 - ingredient
      cocktail312 - cocktail
      dispenser419 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker195)
  (ontable shot446)
  (ontable shot121)
  (ontable shot309)
  (ontable shot30)
  (dispenses dispenser419 ingredient151)
  (dispenses dispenser193 ingredient270)
  (clean shaker195)
  (clean shot446)
  (clean shot121)
  (clean shot309)
  (clean shot30)
  (empty shaker195)
  (empty shot446)
  (empty shot121)
  (empty shot309)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker195 l0)
  (shaker-level shaker195 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient151)
  (cocktail-part2 cocktail312 ingredient270)
)
 (:goal
  (and
      (contains shot446 cocktail312)
      (contains shot121 cocktail312)
      (contains shot309 ingredient270)
)))
