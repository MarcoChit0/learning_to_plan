; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 711189

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 - child
    bread103 - bread-portion
    content213 - content-portion
    tray58 tray326 tray91 - tray
    table173 table302 table361 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray58 kitchen)
     (at tray326 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread103)
     (at_kitchen_content content213)
     (not_allergic_gluten child259)
     (waiting child259 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child259)
    )
  )
)
