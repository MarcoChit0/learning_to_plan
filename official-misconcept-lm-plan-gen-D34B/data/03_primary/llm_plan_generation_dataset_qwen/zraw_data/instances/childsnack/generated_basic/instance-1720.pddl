; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 363181

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child243 - child
    bread277 - bread-portion
    content268 - content-portion
    tray248 tray147 tray497 - tray
    table153 table324 table187 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray147 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_content content268)
     (not_allergic_gluten child243)
     (waiting child243 table324)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child243)
    )
  )
)
