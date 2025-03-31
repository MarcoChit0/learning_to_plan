; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 527747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child452 - child
    bread141 bread286 - bread-portion
    content386 content448 - content-portion
    tray381 tray296 - tray
    table172 table248 table223 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray296 kitchen)
     (at_kitchen_bread bread141)
     (at_kitchen_bread bread286)
     (at_kitchen_content content386)
     (at_kitchen_content content448)
     (no_gluten_bread bread141)
     (no_gluten_content content448)
     (allergic_gluten child452)
     (not_allergic_gluten child157)
     (waiting child157 table223)
     (waiting child452 table172)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child157)
     (served child452)
    )
  )
)
