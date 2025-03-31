; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 931401

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 child362 - child
    bread238 bread329 - bread-portion
    content498 content413 - content-portion
    tray472 tray335 tray150 - tray
    table413 table199 table310 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray472 kitchen)
     (at tray335 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread329)
     (at_kitchen_content content498)
     (at_kitchen_content content413)
     (not_allergic_gluten child362)
     (not_allergic_gluten child221)
     (waiting child221 table199)
     (waiting child362 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child221)
     (served child362)
    )
  )
)
