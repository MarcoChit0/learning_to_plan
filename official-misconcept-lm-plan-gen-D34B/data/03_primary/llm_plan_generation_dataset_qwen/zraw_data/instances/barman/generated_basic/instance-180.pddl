(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot410 shot103 shot55 - shot
      ingredient321 ingredient40 ingredient437 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser301 dispenser20 dispenser28 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot410)
  (ontable shot103)
  (ontable shot55)
  (dispenses dispenser301 ingredient321)
  (dispenses dispenser20 ingredient40)
  (dispenses dispenser28 ingredient437)
  (dispenses dispenser40 ingredient463)
  (clean shaker266)
  (clean shot410)
  (clean shot103)
  (clean shot55)
  (empty shaker266)
  (empty shot410)
  (empty shot103)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient321)
  (cocktail-part2 cocktail1 ingredient463)
)
 (:goal
  (and
      (contains shot410 cocktail1)
      (contains shot103 cocktail1)
)))
