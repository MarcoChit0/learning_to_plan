; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 940893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child388 - child
    bread382 bread189 - bread-portion
    content450 content498 - content-portion
    tray275 tray78 tray83 tray265 - tray
    table135 table286 table426 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray275 kitchen)
     (at tray78 kitchen)
     (at tray83 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread189)
     (at_kitchen_content content450)
     (at_kitchen_content content498)
     (no_gluten_bread bread382)
     (no_gluten_content content498)
     (allergic_gluten child121)
     (not_allergic_gluten child388)
     (waiting child121 table286)
     (waiting child388 table135)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child121)
     (served child388)
    )
  )
)
