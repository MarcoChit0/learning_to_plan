; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 863342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child153 - child
    bread353 - bread-portion
    content42 - content-portion
    tray86 tray254 tray201 - tray
    table483 table106 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at tray254 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content42)
     (not_allergic_gluten child153)
     (waiting child153 table263)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child153)
    )
  )
)
