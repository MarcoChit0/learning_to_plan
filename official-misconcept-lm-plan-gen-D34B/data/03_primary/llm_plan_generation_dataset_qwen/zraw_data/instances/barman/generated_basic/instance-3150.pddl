(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot254 shot257 shot377 - shot
      ingredient50 ingredient219 ingredient341 ingredient290 - ingredient
      cocktail302 - cocktail
      dispenser362 dispenser203 dispenser485 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot254)
  (ontable shot257)
  (ontable shot377)
  (dispenses dispenser362 ingredient50)
  (dispenses dispenser203 ingredient219)
  (dispenses dispenser485 ingredient341)
  (dispenses dispenser78 ingredient290)
  (clean shaker162)
  (clean shot254)
  (clean shot257)
  (clean shot377)
  (empty shaker162)
  (empty shot254)
  (empty shot257)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient341)
  (cocktail-part2 cocktail302 ingredient219)
)
 (:goal
  (and
      (contains shot254 cocktail302)
      (contains shot257 cocktail302)
)))
