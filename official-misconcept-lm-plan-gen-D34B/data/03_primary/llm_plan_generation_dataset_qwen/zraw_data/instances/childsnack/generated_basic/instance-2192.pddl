; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 391910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 - child
    bread191 - bread-portion
    content437 - content-portion
    tray469 - tray
    table86 table90 table172 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_content content437)
     (not_allergic_gluten child352)
     (waiting child352 table86)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child352)
    )
  )
)
