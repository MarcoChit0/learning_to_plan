; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 116017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child264 child414 - child
    bread457 bread12 - bread-portion
    content77 content103 - content-portion
    tray125 tray264 tray59 tray239 - tray
    table317 table189 table183 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at tray264 kitchen)
     (at tray59 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_bread bread12)
     (at_kitchen_content content77)
     (at_kitchen_content content103)
     (no_gluten_bread bread457)
     (no_gluten_content content103)
     (allergic_gluten child414)
     (not_allergic_gluten child264)
     (waiting child264 table183)
     (waiting child414 table317)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child264)
     (served child414)
    )
  )
)
