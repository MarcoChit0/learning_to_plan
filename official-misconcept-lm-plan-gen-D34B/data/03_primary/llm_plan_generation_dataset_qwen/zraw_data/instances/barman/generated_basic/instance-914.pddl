(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot115 shot268 - shot
      ingredient15 ingredient92 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser386 dispenser333 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot115)
  (ontable shot268)
  (dispenses dispenser386 ingredient15)
  (dispenses dispenser333 ingredient92)
  (dispenses dispenser414 ingredient321)
  (clean shaker489)
  (clean shot115)
  (clean shot268)
  (empty shaker489)
  (empty shot115)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient92)
  (cocktail-part2 cocktail1 ingredient321)
)
 (:goal
  (and
      (contains shot115 cocktail1)
)))
