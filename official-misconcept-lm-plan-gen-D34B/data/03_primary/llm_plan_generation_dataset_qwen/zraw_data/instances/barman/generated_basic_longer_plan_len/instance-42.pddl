(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot40 shot327 shot449 - shot
      ingredient483 ingredient195 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser61 dispenser190 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot40)
  (ontable shot327)
  (ontable shot449)
  (dispenses dispenser61 ingredient483)
  (dispenses dispenser190 ingredient195)
  (clean shaker330)
  (clean shot40)
  (clean shot327)
  (clean shot449)
  (empty shaker330)
  (empty shot40)
  (empty shot327)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient483)
  (cocktail-part2 cocktail1 ingredient195)
  (cocktail-part1 cocktail2 ingredient483)
  (cocktail-part2 cocktail2 ingredient195)
)
 (:goal
  (and
      (contains shot40 cocktail2)
      (contains shot327 cocktail1)
)))
