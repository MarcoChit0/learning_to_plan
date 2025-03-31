(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot61 - shot
      ingredient377 ingredient160 ingredient122 - ingredient
      cocktail1 - cocktail
      dispenser326 dispenser492 dispenser435 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot61)
  (dispenses dispenser326 ingredient377)
  (dispenses dispenser492 ingredient160)
  (dispenses dispenser435 ingredient122)
  (clean shaker276)
  (clean shot61)
  (empty shaker276)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient160)
  (cocktail-part2 cocktail1 ingredient377)
)
 (:goal
  (and
      (contains shot61 cocktail1)
)))
