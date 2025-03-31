(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot264 shot0 - shot
      ingredient26 ingredient494 ingredient453 ingredient274 - ingredient
      cocktail1 - cocktail
      dispenser50 dispenser485 dispenser372 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot264)
  (ontable shot0)
  (dispenses dispenser50 ingredient26)
  (dispenses dispenser485 ingredient494)
  (dispenses dispenser372 ingredient453)
  (dispenses dispenser278 ingredient274)
  (clean shaker190)
  (clean shot264)
  (clean shot0)
  (empty shaker190)
  (empty shot264)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient274)
  (cocktail-part2 cocktail1 ingredient494)
)
 (:goal
  (and
      (contains shot264 cocktail1)
)))
