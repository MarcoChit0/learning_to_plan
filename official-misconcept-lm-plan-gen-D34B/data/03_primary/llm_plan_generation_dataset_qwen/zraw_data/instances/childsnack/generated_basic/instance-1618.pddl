; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 601893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 - child
    bread229 - bread-portion
    content425 - content-portion
    tray80 tray192 tray240 tray344 - tray
    table450 table316 table455 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray192 kitchen)
     (at tray240 kitchen)
     (at tray344 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_content content425)
     (not_allergic_gluten child24)
     (waiting child24 table455)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child24)
    )
  )
)
