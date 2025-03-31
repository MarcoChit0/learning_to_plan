; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 651144

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child22 - child
    bread349 - bread-portion
    content2 - content-portion
    tray245 tray37 - tray
    table51 table381 table461 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray37 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_content content2)
     (not_allergic_gluten child22)
     (waiting child22 table461)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child22)
    )
  )
)
