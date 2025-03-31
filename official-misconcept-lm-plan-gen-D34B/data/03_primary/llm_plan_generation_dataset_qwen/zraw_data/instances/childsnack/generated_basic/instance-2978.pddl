; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 582496

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 - child
    bread115 - bread-portion
    content51 - content-portion
    tray127 tray217 tray29 tray4 - tray
    table37 table102 table216 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray127 kitchen)
     (at tray217 kitchen)
     (at tray29 kitchen)
     (at tray4 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_content content51)
     (not_allergic_gluten child4)
     (waiting child4 table216)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child4)
    )
  )
)
