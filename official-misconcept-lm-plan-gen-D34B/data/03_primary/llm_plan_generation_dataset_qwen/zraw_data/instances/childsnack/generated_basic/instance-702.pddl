; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 826649

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child496 - child
    bread211 - bread-portion
    content248 - content-portion
    tray65 tray77 tray149 tray249 - tray
    table383 table347 table249 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray77 kitchen)
     (at tray149 kitchen)
     (at tray249 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_content content248)
     (not_allergic_gluten child496)
     (waiting child496 table347)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child496)
    )
  )
)
