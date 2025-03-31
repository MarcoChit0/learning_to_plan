; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 53915

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child222 - child
    bread337 - bread-portion
    content473 - content-portion
    tray434 tray75 tray245 tray214 - tray
    table159 table294 table300 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray75 kitchen)
     (at tray245 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread337)
     (at_kitchen_content content473)
     (not_allergic_gluten child222)
     (waiting child222 table159)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child222)
    )
  )
)
