; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 479189

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child391 child29 - child
    bread101 bread416 - bread-portion
    content463 content126 - content-portion
    tray331 tray328 tray95 - tray
    table218 table426 table250 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray328 kitchen)
     (at tray95 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_bread bread416)
     (at_kitchen_content content463)
     (at_kitchen_content content126)
     (not_allergic_gluten child391)
     (not_allergic_gluten child29)
     (waiting child391 table250)
     (waiting child29 table426)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child391)
     (served child29)
    )
  )
)
