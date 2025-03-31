(define (problem prob)
 (:domain barman)
 (:objects 
      shaker348 - shaker
      left right - hand
      shot268 - shot
      ingredient161 ingredient367 - ingredient
      cocktail117 - cocktail
      dispenser40 dispenser238 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker348)
  (ontable shot268)
  (dispenses dispenser40 ingredient161)
  (dispenses dispenser238 ingredient367)
  (clean shaker348)
  (clean shot268)
  (empty shaker348)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker348 l0)
  (shaker-level shaker348 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail117 ingredient367)
  (cocktail-part2 cocktail117 ingredient161)
)
 (:goal
  (and
      (contains shot268 cocktail117)
)))
