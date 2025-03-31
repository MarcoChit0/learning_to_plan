(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot41 shot141 shot115 - shot
      ingredient20 ingredient289 ingredient114 ingredient186 - ingredient
      cocktail357 - cocktail
      dispenser206 dispenser95 dispenser336 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot41)
  (ontable shot141)
  (ontable shot115)
  (dispenses dispenser206 ingredient20)
  (dispenses dispenser95 ingredient289)
  (dispenses dispenser336 ingredient114)
  (dispenses dispenser213 ingredient186)
  (clean shaker188)
  (clean shot41)
  (clean shot141)
  (clean shot115)
  (empty shaker188)
  (empty shot41)
  (empty shot141)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient289)
  (cocktail-part2 cocktail357 ingredient114)
)
 (:goal
  (and
      (contains shot41 cocktail357)
      (contains shot141 ingredient289)
)))
