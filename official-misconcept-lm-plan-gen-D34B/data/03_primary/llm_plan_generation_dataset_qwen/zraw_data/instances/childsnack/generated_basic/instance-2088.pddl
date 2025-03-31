; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 178780

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 - child
    bread213 - bread-portion
    content7 - content-portion
    tray207 tray105 tray174 tray77 - tray
    table198 table313 table97 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray105 kitchen)
     (at tray174 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread213)
     (at_kitchen_content content7)
     (not_allergic_gluten child128)
     (waiting child128 table198)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child128)
    )
  )
)
