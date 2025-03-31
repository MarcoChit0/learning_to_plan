(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot444 shot347 - shot
      ingredient289 ingredient276 - ingredient
      cocktail1 - cocktail
      dispenser248 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot444)
  (ontable shot347)
  (dispenses dispenser248 ingredient289)
  (dispenses dispenser97 ingredient276)
  (clean shaker440)
  (clean shot444)
  (clean shot347)
  (empty shaker440)
  (empty shot444)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient276)
  (cocktail-part2 cocktail1 ingredient289)
)
 (:goal
  (and
      (contains shot444 cocktail1)
)))
