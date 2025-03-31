; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 796633

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 - child
    bread57 - bread-portion
    content14 - content-portion
    tray263 tray331 tray208 tray14 - tray
    table74 table157 table235 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray263 kitchen)
     (at tray331 kitchen)
     (at tray208 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_content content14)
     (not_allergic_gluten child176)
     (waiting child176 table235)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child176)
    )
  )
)
