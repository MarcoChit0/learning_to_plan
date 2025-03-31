; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 667286

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread185 - bread-portion
    content311 - content-portion
    tray305 tray32 tray376 - tray
    table230 table207 table432 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray32 kitchen)
     (at tray376 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content311)
     (not_allergic_gluten child372)
     (waiting child372 table432)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
