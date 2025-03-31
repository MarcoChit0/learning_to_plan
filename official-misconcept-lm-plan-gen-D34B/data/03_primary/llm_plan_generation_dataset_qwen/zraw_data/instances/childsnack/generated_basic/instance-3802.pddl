; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 429042

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child207 - child
    bread25 - bread-portion
    content238 - content-portion
    tray214 - tray
    table383 table483 table155 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray214 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_content content238)
     (not_allergic_gluten child207)
     (waiting child207 table483)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child207)
    )
  )
)
