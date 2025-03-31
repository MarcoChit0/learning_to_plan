(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot121 shot308 shot489 - shot
      ingredient29 ingredient202 ingredient192 ingredient278 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser402 dispenser435 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot121)
  (ontable shot308)
  (ontable shot489)
  (dispenses dispenser110 ingredient29)
  (dispenses dispenser402 ingredient202)
  (dispenses dispenser435 ingredient192)
  (dispenses dispenser446 ingredient278)
  (clean shaker113)
  (clean shot121)
  (clean shot308)
  (clean shot489)
  (empty shaker113)
  (empty shot121)
  (empty shot308)
  (empty shot489)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient202)
  (cocktail-part2 cocktail1 ingredient278)
)
 (:goal
  (and
      (contains shot121 cocktail1)
      (contains shot308 cocktail1)
)))
