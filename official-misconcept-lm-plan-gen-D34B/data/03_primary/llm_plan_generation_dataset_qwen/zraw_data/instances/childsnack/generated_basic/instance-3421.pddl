; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 344604

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 - child
    bread443 - bread-portion
    content463 - content-portion
    tray93 tray415 tray393 tray439 - tray
    table441 table496 table146 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray415 kitchen)
     (at tray393 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_content content463)
     (not_allergic_gluten child138)
     (waiting child138 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child138)
    )
  )
)
