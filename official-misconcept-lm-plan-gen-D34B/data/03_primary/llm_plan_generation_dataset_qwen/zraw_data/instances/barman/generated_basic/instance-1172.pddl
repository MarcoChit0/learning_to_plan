(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot480 shot314 - shot
      ingredient185 ingredient264 ingredient320 - ingredient
      cocktail1 - cocktail
      dispenser451 dispenser349 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot480)
  (ontable shot314)
  (dispenses dispenser451 ingredient185)
  (dispenses dispenser349 ingredient264)
  (dispenses dispenser412 ingredient320)
  (clean shaker266)
  (clean shot480)
  (clean shot314)
  (empty shaker266)
  (empty shot480)
  (empty shot314)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient320)
  (cocktail-part2 cocktail1 ingredient185)
)
 (:goal
  (and
      (contains shot480 cocktail1)
)))
