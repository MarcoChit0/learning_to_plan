; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 194231

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 - child
    bread298 - bread-portion
    content319 - content-portion
    tray226 tray251 tray50 tray392 - tray
    table276 table400 table392 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray251 kitchen)
     (at tray50 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_content content319)
     (not_allergic_gluten child179)
     (waiting child179 table392)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child179)
    )
  )
)
