; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 294940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child199 - child
    bread298 - bread-portion
    content146 - content-portion
    tray136 tray483 - tray
    table271 table152 table254 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray136 kitchen)
     (at tray483 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_content content146)
     (not_allergic_gluten child199)
     (waiting child199 table152)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child199)
    )
  )
)
