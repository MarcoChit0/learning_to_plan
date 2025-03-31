; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 610588

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 child19 - child
    bread189 bread114 - bread-portion
    content349 content395 - content-portion
    tray105 tray162 tray75 - tray
    table318 table289 table432 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray162 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread189)
     (at_kitchen_bread bread114)
     (at_kitchen_content content349)
     (at_kitchen_content content395)
     (no_gluten_bread bread189)
     (no_gluten_content content395)
     (allergic_gluten child97)
     (not_allergic_gluten child19)
     (waiting child97 table318)
     (waiting child19 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child97)
     (served child19)
    )
  )
)
