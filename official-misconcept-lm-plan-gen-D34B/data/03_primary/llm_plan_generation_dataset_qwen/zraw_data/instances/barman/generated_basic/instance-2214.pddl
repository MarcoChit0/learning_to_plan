(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot234 shot447 shot407 - shot
      ingredient45 ingredient236 ingredient423 ingredient295 - ingredient
      cocktail277 - cocktail
      dispenser445 dispenser495 dispenser473 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot234)
  (ontable shot447)
  (ontable shot407)
  (dispenses dispenser445 ingredient45)
  (dispenses dispenser495 ingredient236)
  (dispenses dispenser473 ingredient423)
  (dispenses dispenser45 ingredient295)
  (clean shaker185)
  (clean shot234)
  (clean shot447)
  (clean shot407)
  (empty shaker185)
  (empty shot234)
  (empty shot447)
  (empty shot407)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient295)
  (cocktail-part2 cocktail277 ingredient423)
)
 (:goal
  (and
      (contains shot234 cocktail277)
      (contains shot447 ingredient423)
)))
