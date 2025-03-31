; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 172416

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 - child
    bread490 - bread-portion
    content250 - content-portion
    tray252 tray463 tray361 - tray
    table58 table387 table413 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray463 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread490)
     (at_kitchen_content content250)
     (not_allergic_gluten child136)
     (waiting child136 table387)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child136)
    )
  )
)
