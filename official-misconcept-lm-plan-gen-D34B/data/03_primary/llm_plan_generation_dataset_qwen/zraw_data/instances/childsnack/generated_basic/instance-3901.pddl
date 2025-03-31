; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 842913

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 - child
    bread371 - bread-portion
    content20 - content-portion
    tray213 tray343 - tray
    table190 table57 table21 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray213 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread371)
     (at_kitchen_content content20)
     (not_allergic_gluten child396)
     (waiting child396 table190)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child396)
    )
  )
)
