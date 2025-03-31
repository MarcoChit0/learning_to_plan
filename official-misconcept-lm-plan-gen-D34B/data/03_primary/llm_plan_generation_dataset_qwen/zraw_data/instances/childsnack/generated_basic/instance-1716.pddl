; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 705376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child195 - child
    bread395 - bread-portion
    content167 - content-portion
    tray254 tray407 tray56 - tray
    table181 table471 table365 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray254 kitchen)
     (at tray407 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread395)
     (at_kitchen_content content167)
     (not_allergic_gluten child195)
     (waiting child195 table471)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child195)
    )
  )
)
