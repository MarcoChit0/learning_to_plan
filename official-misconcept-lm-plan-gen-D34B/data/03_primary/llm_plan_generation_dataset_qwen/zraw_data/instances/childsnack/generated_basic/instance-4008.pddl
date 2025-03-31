; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 451255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 - child
    bread365 - bread-portion
    content80 - content-portion
    tray35 tray178 tray263 - tray
    table290 table340 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray178 kitchen)
     (at tray263 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_content content80)
     (not_allergic_gluten child271)
     (waiting child271 table340)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child271)
    )
  )
)
