(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot313 shot283 shot51 - shot
      ingredient51 ingredient378 - ingredient
      cocktail153 - cocktail
      dispenser442 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot313)
  (ontable shot283)
  (ontable shot51)
  (dispenses dispenser442 ingredient51)
  (dispenses dispenser487 ingredient378)
  (clean shaker157)
  (clean shot313)
  (clean shot283)
  (clean shot51)
  (empty shaker157)
  (empty shot313)
  (empty shot283)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient51)
  (cocktail-part2 cocktail153 ingredient378)
)
 (:goal
  (and
      (contains shot313 cocktail153)
      (contains shot283 cocktail153)
)))
