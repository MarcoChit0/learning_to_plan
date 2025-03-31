; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 218314

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 - child
    bread390 - bread-portion
    content416 - content-portion
    tray84 tray425 tray324 tray469 - tray
    table267 table229 table65 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray425 kitchen)
     (at tray324 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread390)
     (at_kitchen_content content416)
     (not_allergic_gluten child433)
     (waiting child433 table229)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child433)
    )
  )
)
