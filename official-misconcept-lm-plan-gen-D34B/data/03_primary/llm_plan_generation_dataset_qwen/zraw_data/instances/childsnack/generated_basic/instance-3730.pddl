; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 287113

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 - child
    bread190 - bread-portion
    content419 - content-portion
    tray183 tray148 tray492 - tray
    table121 table371 table297 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray183 kitchen)
     (at tray148 kitchen)
     (at tray492 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_content content419)
     (not_allergic_gluten child137)
     (waiting child137 table121)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child137)
    )
  )
)
