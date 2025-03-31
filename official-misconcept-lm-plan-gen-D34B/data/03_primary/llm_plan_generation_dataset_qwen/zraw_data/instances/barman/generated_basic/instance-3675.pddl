(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot440 shot188 shot366 - shot
      ingredient31 ingredient96 ingredient271 ingredient170 - ingredient
      cocktail20 - cocktail
      dispenser386 dispenser117 dispenser49 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot440)
  (ontable shot188)
  (ontable shot366)
  (dispenses dispenser386 ingredient31)
  (dispenses dispenser117 ingredient96)
  (dispenses dispenser49 ingredient271)
  (dispenses dispenser85 ingredient170)
  (clean shaker317)
  (clean shot440)
  (clean shot188)
  (clean shot366)
  (empty shaker317)
  (empty shot440)
  (empty shot188)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail20 ingredient31)
  (cocktail-part2 cocktail20 ingredient96)
)
 (:goal
  (and
      (contains shot440 cocktail20)
      (contains shot188 ingredient31)
)))
