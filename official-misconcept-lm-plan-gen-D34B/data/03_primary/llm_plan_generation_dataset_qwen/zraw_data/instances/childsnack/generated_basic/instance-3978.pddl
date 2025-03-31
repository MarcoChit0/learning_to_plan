; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 656737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child41 - child
    bread399 - bread-portion
    content364 - content-portion
    tray70 tray77 tray277 - tray
    table435 table378 table362 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray77 kitchen)
     (at tray277 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_content content364)
     (not_allergic_gluten child41)
     (waiting child41 table435)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child41)
    )
  )
)
