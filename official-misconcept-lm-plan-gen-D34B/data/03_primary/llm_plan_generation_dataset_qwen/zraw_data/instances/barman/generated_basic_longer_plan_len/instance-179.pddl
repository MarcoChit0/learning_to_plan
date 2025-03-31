(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot122 shot329 shot169 shot14 - shot
      ingredient404 ingredient406 - ingredient
      cocktail1 - cocktail
      dispenser487 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot122)
  (ontable shot329)
  (ontable shot169)
  (ontable shot14)
  (dispenses dispenser487 ingredient404)
  (dispenses dispenser467 ingredient406)
  (clean shaker318)
  (clean shot122)
  (clean shot329)
  (clean shot169)
  (clean shot14)
  (empty shaker318)
  (empty shot122)
  (empty shot329)
  (empty shot169)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient404)
  (cocktail-part2 cocktail1 ingredient406)
)
 (:goal
  (and
      (contains shot122 cocktail1)
      (contains shot329 ingredient404)
      (contains shot169 ingredient404)
)))
