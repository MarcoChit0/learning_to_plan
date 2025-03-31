; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 653165

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 - child
    bread251 - bread-portion
    content88 - content-portion
    tray146 tray171 tray124 tray411 - tray
    table488 table332 table494 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray171 kitchen)
     (at tray124 kitchen)
     (at tray411 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_content content88)
     (not_allergic_gluten child152)
     (waiting child152 table332)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child152)
    )
  )
)
