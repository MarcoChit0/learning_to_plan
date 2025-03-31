; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 65516

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child230 - child
    bread399 - bread-portion
    content58 - content-portion
    tray338 tray238 - tray
    table161 table28 table131 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at tray238 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_content content58)
     (not_allergic_gluten child230)
     (waiting child230 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child230)
    )
  )
)
