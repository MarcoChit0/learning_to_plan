; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 527925

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child118 - child
    bread400 - bread-portion
    content101 - content-portion
    tray6 tray277 tray35 - tray
    table159 table346 table7 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray277 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content101)
     (not_allergic_gluten child118)
     (waiting child118 table7)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child118)
    )
  )
)
