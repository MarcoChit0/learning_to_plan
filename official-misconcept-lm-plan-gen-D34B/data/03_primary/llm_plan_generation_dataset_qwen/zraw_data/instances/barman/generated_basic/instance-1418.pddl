(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot269 shot379 - shot
      ingredient443 ingredient290 - ingredient
      cocktail157 - cocktail
      dispenser184 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot269)
  (ontable shot379)
  (dispenses dispenser184 ingredient443)
  (dispenses dispenser483 ingredient290)
  (clean shaker157)
  (clean shot269)
  (clean shot379)
  (empty shaker157)
  (empty shot269)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient290)
  (cocktail-part2 cocktail157 ingredient443)
)
 (:goal
  (and
      (contains shot269 cocktail157)
)))
