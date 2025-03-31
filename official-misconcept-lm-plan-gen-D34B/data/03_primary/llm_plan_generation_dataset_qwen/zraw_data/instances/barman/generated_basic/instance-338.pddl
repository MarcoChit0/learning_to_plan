(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot223 - shot
      ingredient111 ingredient161 - ingredient
      cocktail1 - cocktail
      dispenser102 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot223)
  (dispenses dispenser102 ingredient111)
  (dispenses dispenser361 ingredient161)
  (clean shaker87)
  (clean shot223)
  (empty shaker87)
  (empty shot223)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient111)
  (cocktail-part2 cocktail1 ingredient161)
)
 (:goal
  (and
      (contains shot223 cocktail1)
)))
