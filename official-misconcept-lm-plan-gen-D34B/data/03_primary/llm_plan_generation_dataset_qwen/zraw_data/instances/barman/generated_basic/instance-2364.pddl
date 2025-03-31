(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot124 - shot
      ingredient270 ingredient278 ingredient403 ingredient251 - ingredient
      cocktail440 - cocktail
      dispenser435 dispenser235 dispenser317 dispenser485 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot124)
  (dispenses dispenser435 ingredient270)
  (dispenses dispenser235 ingredient278)
  (dispenses dispenser317 ingredient403)
  (dispenses dispenser485 ingredient251)
  (clean shaker139)
  (clean shot124)
  (empty shaker139)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail440 ingredient251)
  (cocktail-part2 cocktail440 ingredient278)
)
 (:goal
  (and
      (contains shot124 cocktail440)
)))
