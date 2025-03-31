; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 475837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child172 - child
    bread341 bread108 - bread-portion
    content145 content319 - content-portion
    tray171 tray7 tray420 tray125 - tray
    table27 table385 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray171 kitchen)
     (at tray7 kitchen)
     (at tray420 kitchen)
     (at tray125 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_bread bread108)
     (at_kitchen_content content145)
     (at_kitchen_content content319)
     (no_gluten_bread bread341)
     (no_gluten_content content145)
     (allergic_gluten child416)
     (not_allergic_gluten child172)
     (waiting child416 table27)
     (waiting child172 table324)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child416)
     (served child172)
    )
  )
)
