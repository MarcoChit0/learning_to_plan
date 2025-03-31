(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot60 shot414 shot422 - shot
      ingredient98 ingredient51 ingredient190 - ingredient
      cocktail25 - cocktail
      dispenser11 dispenser319 dispenser439 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot60)
  (ontable shot414)
  (ontable shot422)
  (dispenses dispenser11 ingredient98)
  (dispenses dispenser319 ingredient51)
  (dispenses dispenser439 ingredient190)
  (clean shaker76)
  (clean shot60)
  (clean shot414)
  (clean shot422)
  (empty shaker76)
  (empty shot60)
  (empty shot414)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail25 ingredient98)
  (cocktail-part2 cocktail25 ingredient51)
)
 (:goal
  (and
      (contains shot60 cocktail25)
      (contains shot414 cocktail25)
)))
